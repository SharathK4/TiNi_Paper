import { Component, type ReactNode } from 'react'

/* A rendering error in one section must not blank the whole route. */
export class ErrorBoundary extends Component<{ name: string; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <section className="panel">
          <div className="panel__index">!</div>
          <h2 className="panel__title">{this.props.name}</h2>
          <p className="panel__lead">This section could not be drawn: {this.state.error.message}. The rest of the page is unaffected.</p>
        </section>
      )
    }
    return this.props.children
  }
}
